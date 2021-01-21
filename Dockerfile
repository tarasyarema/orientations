FROM sagemath/sagemath:latest 

WORKDIR /home/sage/orientations

COPY src .

CMD ["sage"]
