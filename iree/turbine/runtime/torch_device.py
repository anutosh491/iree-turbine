# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""PyTorch tensors backed by an asynchronous IREE HAL device.

The optional adapter uses PyTorch's PrivateUse1 registration. GPU calls submit
work synchronously and return tensors with pending completion fences. Host
readback is explicit: ``await device.to_cpu(tensor)`` yields to the event loop.
"""

import asyncio
from dataclasses import dataclass

import numpy as np
import torch
import iree.runtime as rt

from ..support.conversions import dtype_to_element_type, torch_dtype_to_numpy

_active_device = None
_registered = False
_device_name = None
_storages = {}


@dataclass
class _Storage:
    owner: "TorchDevice"
    array: rt.DeviceArray
    completion: object = None


def _storage(tensor):
    entry = _storages.get(tensor.untyped_storage()._cdata)
    if entry is None:
        raise TypeError("Expected a tensor owned by the IREE device adapter")
    _, storage = entry
    storage.owner._check()
    return storage


def _wrap(storage):
    dtype = torch.from_numpy(np.empty((), dtype=storage.array.dtype)).dtype
    tensor = torch._C._acc.create_empty_tensor(storage.array.shape, dtype)
    native_storage = tensor.untyped_storage()
    reference = native_storage._weak_ref()
    _storages[native_storage._cdata] = reference, storage
    return tensor


def _collect_storages(owner=None):
    # Native detach/alias operations share StorageImpl. Its weak reference
    # keeps the sidecar alive until all tensor aliases have been released.
    for key, (reference, storage) in list(_storages.items()):
        if torch.UntypedStorage._expired(reference) or storage.owner is owner:
            del _storages[key]
            torch.UntypedStorage._free_weak_ref(reference)


def _empty_strided(
    size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
):
    expected = torch.empty(size, device="meta").stride()
    if tuple(stride) != expected:
        raise NotImplementedError(
            "IREE tensor allocation currently requires contiguous strides"
        )
    return _empty(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


def _empty(
    size, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    if _active_device is None:
        raise RuntimeError("Create a TorchDevice before allocating IREE tensors")
    return _active_device.empty(size, dtype=dtype or torch.float32)


def _copy_from(source, target, non_blocking=False):
    if target.device.type != _device_name:
        raise RuntimeError(
            "IREE host readback is asynchronous; use await device.to_cpu(tensor)"
        )
    storage = _storage(target)
    if tuple(source.shape) != tuple(target.shape) or source.dtype != target.dtype:
        raise ValueError("IREE tensor copy requires matching shapes and dtypes")
    if source.device.type == _device_name:
        source_storage = _storage(source)
        if source_storage.owner is not storage.owner:
            raise ValueError("Device tensor copy requires the same IREE device")
        if (
            source_storage.array._buffer_view.get_buffer()
            == storage.array._buffer_view.get_buffer()
        ):
            return target
        _enqueue(
            storage,
            storage.owner.hal_device.queue_copy,
            (
                source_storage.array._buffer_view.get_buffer(),
                storage.array._buffer_view.get_buffer(),
            ),
            (source_storage,),
        )
        return target
    if source.device.type != "cpu":
        raise ValueError("Tensor copy requires a CPU or IREE tensor")
    completion = storage.owner.session.enqueue_upload(
        source.detach().contiguous().numpy().copy()
    )
    storage.owner._pending.append(completion)
    storage.array = completion.value
    storage.completion = completion
    return target


def _enqueue(storage, operation, arguments, dependencies=()):
    owner = storage.owner
    storages = (storage, *dependencies)
    fences = [s.completion.fence for s in storages if s.completion is not None]
    semaphore = owner.hal_device.create_semaphore(0)
    operation(
        *arguments,
        wait_semaphores=(
            rt.HalFence.join(fences) if fences else rt.HalFence.create_at(semaphore, 0)
        ),
        signal_semaphores=rt.HalFence.create_at(semaphore, 1),
    )
    result = rt.AsyncResult(
        owner.session,
        storage.array,
        semaphore,
        tuple(s.array for s in storages) + arguments,
    )
    owner._pending.append(result)
    storage.completion = result


def _fill(tensor, value):
    storage = _storage(tensor)
    pattern = np.asarray([value], dtype=storage.array.dtype)
    command = rt.HalCommandBuffer(storage.owner.hal_device)
    command.fill(storage.array._buffer_view.get_buffer(), pattern, end=True)
    _enqueue(storage, storage.owner.hal_device.queue_execute, (command,))
    return tensor


def _copy(target, source, non_blocking=False):
    return _copy_from(source, target, non_blocking)


def _zero(tensor):
    return _fill(tensor, 0)


def _register(name):
    global _registered, _device_name
    if _registered:
        if name != _device_name:
            raise RuntimeError("The IREE PyTorch device name is already registered")
        return
    from torch.utils.backend_registration import _setup_privateuseone_for_python_backend

    _setup_privateuseone_for_python_backend(name)
    _device_name = name
    for name, implementation in (
        ("empty.memory_format", _empty),
        ("empty_strided", _empty_strided),
        ("detach", torch.library.fallthrough_kernel),
        ("alias", torch.library.fallthrough_kernel),
        ("_copy_from", _copy_from),
        ("copy_", _copy),
        ("fill_.Scalar", _fill),
        ("zero_", _zero),
    ):
        if implementation is not torch.library.fallthrough_kernel:
            # Device kernels submit runtime work; Dynamo must not trace their
            # Python allocation and completion bookkeeping as model code.
            implementation = torch.compiler.disable(implementation)
        torch.library.impl("aten::" + name, "PrivateUse1")(implementation)
    _registered = True


class TorchDevice:
    """Owns one IREE-backed PyTorch device and its async completion session.

    Construct inside a running event loop with an initialized HAL device.
    Only one adapter may be active. The first implementation supports dense,
    contiguous tensors, compiled graphs, and explicit asynchronous readback.
    It does not provide eager operator fallback or device random generators.
    """

    def __init__(self, hal_device, *, target="webgpu", name="iree"):
        global _active_device
        if _active_device is not None:
            raise RuntimeError("Only one IREE PyTorch device may be active")
        self._loop = asyncio.get_running_loop()
        self.session = rt.AsyncSession(hal_device)
        _register(name)
        self.hal_device = hal_device
        self.target = target
        self.torch_device = torch.device(f"{name}:0")
        self._pending = []
        self._closed = False
        _active_device = self

    def _check(self):
        if self._closed:
            raise RuntimeError("TorchDevice is closed")
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("TorchDevice belongs to another event loop")

    def empty(self, shape, *, dtype=torch.float32):
        self._check()
        dtype_np = np.dtype(torch_dtype_to_numpy(dtype))
        buffer = self.hal_device.allocator.allocate_buffer(
            memory_type=int(rt.MemoryType.DEVICE_LOCAL),
            allowed_usage=int(rt.BufferUsage.DEFAULT),
            allocation_size=int(np.prod(shape)) * dtype_np.itemsize,
        )
        view = rt.HalBufferView(buffer, shape, dtype_to_element_type(dtype))
        return _wrap(
            _Storage(
                self, rt.DeviceArray(self.hal_device, view, override_dtype=dtype_np)
            )
        )

    def upload(self, tensor):
        """Queue an owned copy of a CPU tensor and return its device tensor."""
        self._check()
        if tensor.device.type != "cpu":
            raise ValueError("upload() expects a CPU tensor")
        completion = self.session.enqueue_upload(
            tensor.detach().contiguous().numpy().copy()
        )
        self._pending.append(completion)
        return _wrap(_Storage(self, completion.value, completion))

    async def to_cpu(self, tensor):
        """Read a tensor back without blocking the browser event loop."""
        storage = _storage(tensor)
        if storage.owner is not self:
            raise ValueError("Tensor belongs to another IREE device")
        fences = [storage.completion.fence] if storage.completion is not None else []
        value = await self.session.download(storage.array, wait_fences=fences)
        await self.synchronize()
        return torch.from_numpy(value)

    async def synchronize(self):
        """Await submitted work and release completed transfer owners."""
        self._check()
        pending, self._pending = self._pending, []
        if pending:
            await asyncio.gather(*(value.result() for value in pending))
        _collect_storages()

    def backend(self, graph_module, example_inputs):
        """Compiler backend suitable for ``torch.compile(backend=device.backend)``."""
        self._check()
        from ..dynamo.backends.async_backend import make_backend

        return make_backend(self)(graph_module, example_inputs)

    def close(self):
        """Close after awaiting ``synchronize()``."""
        global _active_device
        if self._pending:
            raise RuntimeError("Await synchronize() before closing TorchDevice")
        if not self._closed:
            self.session.close()
            _collect_storages(self)
            self._closed = True
            if _active_device is self:
                _active_device = None
