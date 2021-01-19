FROM sagemath/sagemath:latest 

WORKDIR /tmp

COPY src/lib.sage lib.sage

COPY src/tests.sage tests.sage

CMD ["sage"]
