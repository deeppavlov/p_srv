# p_srv
Paraphraser microservice
```sh
docker run -d -h p_srv                                              \
           --name p_srv                                             \
           -e "AMQP_URI=amqp://user:password@host"                  \
           -v /data/paraphraser:/data                               \
           seliverstov/p_srv:latest

```
