FROM migrate/migrate:4

WORKDIR /migrations

ADD ../diploma-gateway/migrations /migrations

RUN apk add --no-cache bash

add ./migrate-service/start.sh ./start.sh
RUN chmod +x ./start.sh
run cat ./start.sh

#тут нет Make, поэтому все ручками пишем

CMD [ "-path=/migrations", "-database=postgres://postgres_user:superStrongPassword@database:5432/diploma?sslmode=disable","up"]