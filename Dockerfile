FROM sagemath/sagemath:latest 

WORKDIR /tmp

COPY src/lib.sage lib.sage

COPY src/tests.py tests.sage

CMD ["sage", "tests.sage"]
