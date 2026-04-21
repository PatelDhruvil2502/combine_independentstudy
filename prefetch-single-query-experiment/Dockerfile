FROM python:3.10-slim
RUN apt-get update && apt-get install -y build-essential python3-dev git
ARG DISABLE_PREFETCH=0
RUN pip install numpy
RUN git clone --depth 1 https://github.com/nmslib/hnswlib.git /tmp/hnswlib
RUN if [ "$DISABLE_PREFETCH" = "1" ]; then \
      python -c "from pathlib import Path; p=Path('/tmp/hnswlib/hnswlib/hnswalg.h'); t=p.read_text(); marker='#include <memory>'; inject='\\n#ifdef DISABLE_HNSW_PREFETCH\\n#define _mm_prefetch(a, sel) ((void)0)\\n#endif\\n'; t=t.replace(marker, marker+inject, 1) if marker in t and 'DISABLE_HNSW_PREFETCH' not in t else t; p.write_text(t)"; \
      CXXFLAGS='-DDISABLE_HNSW_PREFETCH' pip install /tmp/hnswlib; \
    else \
      pip install /tmp/hnswlib; \
    fi && rm -rf /tmp/hnswlib
COPY worker.py /app/worker.py
WORKDIR /app
CMD ["python", "worker.py"]
