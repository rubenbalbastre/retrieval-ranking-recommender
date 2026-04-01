FROM pgvector/pgvector:pg16

COPY infra/db/init/01_extensions.sql /docker-entrypoint-initdb.d/01_extensions.sql
COPY infra/db/init/02_schema.sql /docker-entrypoint-initdb.d/02_schema.sql
