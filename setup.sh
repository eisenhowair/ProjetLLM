#!/bin/bash

# pour pc de la fac

sudo apt install pip

pip install -r requirements.txt

cat .env

current_path=$(pwd)

path_array=(${current_path//\// })

third_directory=${path_array[2]}

new_command="/home/UHA/${third_directory}/.local/lib/python3.8/site-packages/huggingface-cli login"

$new_command

curl -fsSL https://ollama.com/install.sh | sh
: ' 
mettre à jour chainlit et langchain résoud le problème suivant

Traceback (most recent call last):
  File "dossier_entier_RAG.py", line 22, in <module>
    def auth_callback(username: str, password: str):
  File "/usr/local/lib/python3.8/dist-packages/chainlit/telemetry.py", line 100, in wrapper
    with chainlit_telemetry.tracer.start_as_current_span(
  File "/usr/local/lib/python3.8/dist-packages/chainlit/telemetry.py", line 79, in tracer
    self._tracer = self.configure_tracer()
  File "/usr/local/lib/python3.8/dist-packages/chainlit/telemetry.py", line 53, in configure_tracer
    import uptrace
  File "/usr/local/lib/python3.8/dist-packages/uptrace/__init__.py", line 4, in <module>
    from .uptrace import (
  File "/usr/local/lib/python3.8/dist-packages/uptrace/uptrace.py", line 13, in <module>
    from .traces import configure_traces
  File "/usr/local/lib/python3.8/dist-packages/uptrace/traces.py", line 3, in <module>
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
  File "/usr/local/lib/python3.8/dist-packages/opentelemetry/exporter/otlp/proto/grpc/trace_exporter/__init__.py", line 24, in <module>
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import (
  File "/usr/local/lib/python3.8/dist-packages/opentelemetry/exporter/otlp/proto/common/trace_encoder.py", line 16, in <module>
    from opentelemetry.exporter.otlp.proto.common._internal.trace_encoder import (
  File "/usr/local/lib/python3.8/dist-packages/opentelemetry/exporter/otlp/proto/common/_internal/trace_encoder/__init__.py", line 42, in <module>
    SpanKind.INTERNAL: PB2SPan.SpanKind.SPAN_KIND_INTERNAL,
AttributeError: 'EnumTypeWrapper' object has no attribute 'SPAN_KIND_INTERNAL'
'