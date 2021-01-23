FROM sagemath/sagemath:latest-py3

USER root

RUN apt-get update && apt-get install make && sage -pip install sage-package sphinx

USER sage

COPY --chown=sage:sage . ${HOME}

RUN make test

RUN make install

RUN make docs

CMD ["sage"]
