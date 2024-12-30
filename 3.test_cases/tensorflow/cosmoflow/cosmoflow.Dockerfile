# You must build the base image with ../tensorflow.Dockerfile
FROM tensorflow:latest 

RUN pip3 install wandb
COPY train.py /workspace/train.py
COPY configs /workspace/configs
COPY utils /workspace/utils
COPY data /workspace/data
COPY models /workspace/models

# Modify line 336 in /usr/lib/python3.12/random.py
# https://stackoverflow.com/questions/78498302/keras-3-tensorflow-2-16-and-keras-2-compatibility-problem
RUN sed -i '336s/return self.randrange(a, b+1)/return self.randrange(int(a), int(b+1))/' /usr/lib/python3.12/random.py
# Install MLPerf-logging
RUN pip install --no-cache-dir "git+https://github.com/mlcommons/logging.git"

## Set Open MPI variables to exclude network interface and conduit.
ENV OMPI_MCA_pml=^cm,ucx            \
    OMPI_MCA_btl=tcp,self           \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth_def_agent\
    OPAL_PREFIX=/opt/amazon/openmpi \
    NCCL_SOCKET_IFNAME=^docker,lo,veth_def_agent,eth

#RUN mv $OPEN_MPI_PATH/bin/mpirun $OPEN_MPI_PATH/bin/mpirun.real \
# && echo '#!/bin/bash' > $OPEN_MPI_PATH/bin/mpirun \
# && echo '/opt/amazon/openmpi/bin/mpirun.real "$@"' >> $OPEN_MPI_PATH/bin/mpirun \
# && chmod a+x $OPEN_MPI_PATH/bin/mpirun
### Turn off PMIx Error https://github.com/open-mpi/ompi/issues/7516
#ENV PMIX_MCA_gds=hash
