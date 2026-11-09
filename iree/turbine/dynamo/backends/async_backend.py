# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile captured PyTorch graphs for an asynchronous IREE device."""

import time

import torch
import iree.runtime as rt
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from iree.compiler.api import Output, Session, Source
from iree.compiler import ir

from .base import _import_graph, _restore_none_outputs, logger
from ...runtime.torch_device import _Storage, _storage, _wrap


def make_backend(device):
    def compile_graph(graph_module, example_inputs):
        started = time.perf_counter()
        module_op, output_mask = _import_graph(graph_module, example_inputs)
        if device.target == "webgpu":
            with module_op.context:
                module_op.attributes["stream.resources"] = ir.Attribute.parse(
                    "#stream.resource_config<{max_allocation_size = 9223372036854775807, "
                    "min_buffer_offset_alignment = 256, max_buffer_range = 9223372036854775807, "
                    "min_buffer_range_alignment = 256, index_bits = 64, "
                    "alias_mutable_bindings = true, memory_model = Discrete}>"
                )
        with Session() as session:
            session.set_flags(
                "--iree-input-type=torch",
                f"--iree-hal-target-device={device.target}",
                "--iree-execution-model=async-external",
                "--iree-opt-const-eval=false",
                "--iree-opt-const-expr-hoisting=false",
                "--iree-torch-allow-non-finites=false",
            )
            with session.invocation() as invocation, Source.wrap_buffer(
                session, str(module_op).encode()
            ) as source:
                invocation.enable_console_diagnostics()
                if not invocation.parse_source(source) or not invocation.execute():
                    raise RuntimeError("IREE compilation of the PyTorch graph failed")
                with Output.open_membuffer() as output:
                    invocation.output_vm_bytecode(output)
                    binary = bytes(output.map_memory())
        config = rt.Config(device=device.hal_device)
        context = rt.SystemContext(config=config)
        vm_module = rt.VmModule.copy_buffer(config.vm_instance, binary)
        context.add_vm_module(vm_module)
        function = getattr(getattr(context.modules, vm_module.name), "main$async")
        invocation = device.session.bind(function, abi="coarse-fences")
        logger.debug(
            "Compiled async IREE graph in %.3f s", time.perf_counter() - started
        )

        def execute(*arguments):
            values, fences = [], []
            for argument in arguments:
                if isinstance(argument, torch.Tensor):
                    storage = _storage(argument)
                    if storage.owner is not device:
                        raise ValueError(
                            "Compiled graph input belongs to another device"
                        )
                    values.append(storage.array)
                    if storage.completion is not None:
                        fences.append(storage.completion.fence)
                else:
                    values.append(argument)
            result = invocation.submit(*values, wait_fences=fences)
            device._pending.append(result)
            outputs = result.value
            if not isinstance(outputs, (tuple, list)):
                outputs = (outputs,)
            # Keep the loaded VM context alive with the compiled callable.
            _ = context
            tensors = [_wrap(_Storage(device, array, result)) for array in outputs]
            return _restore_none_outputs(tensors, output_mask)

        return make_boxed_func(execute)

    return aot_autograd(fw_compiler=compile_graph)
