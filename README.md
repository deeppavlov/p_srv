# Paraphraser microservice
```sh
docker run --rm -d -h p_srv.local                                   \
           --name p_srv                                             \
           -e "AMQP_URI=amqp://user:password@host"                  \
           -v /data/paraphraser:/data                               \
           seliverstov/p_srv:latest

```
