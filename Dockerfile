FROM sagemath/sagemath:latest 

WORKDIR /tmp

COPY src .

CMD ["sage"]
