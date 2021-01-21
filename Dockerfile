FROM sagemath/sagemath:latest 

WORKDIR /tmp/orientations

COPY src .

CMD ["sage"]
