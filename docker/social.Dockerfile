# Domain image: social statistics (surveys, education, demography-adjacent)
# Inherits from bench-base and adds domain-specific libraries.
FROM bench-base:v1

RUN pip install \
        pingouin==0.5.4 \
        factor-analyzer==0.5.1 \
        pyreadstat==1.2.7 \
        linearmodels==6.1 \
        econml==0.15.1

WORKDIR /workspace
