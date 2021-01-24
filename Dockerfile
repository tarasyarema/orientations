FROM sagemath/sagemath:latest-py3

USER root

RUN apt-get update && apt-get install make && sage -pip install sage-package sphinx

USER sage

COPY --chown=sage:sage . ${HOME}

RUN make install

RUN make doc

CMD ["sage"]
