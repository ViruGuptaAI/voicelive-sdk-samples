FROM mcr.microsoft.com/azurelinux/base/nodejs:20 AS web

RUN tdnf distro-sync -y && \
    tdnf install -y jq && \
    tdnf clean all

WORKDIR /web
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run lint
RUN npm run build

FROM mcr.microsoft.com/azurelinux/base/python:3 AS final

RUN tdnf distro-sync -y && \
    tdnf install -y ca-certificates && \
    tdnf upgrade -y && \
    tdnf clean all

COPY --from=web /web/out /web/out
COPY app.py /web/app.py

RUN --mount=type=cache,target=/root/.cache/pip pip install aiohttp azure-identity "azure-ai-agents" azure-ai-projects python-dotenv

WORKDIR /web

EXPOSE 3000

ENTRYPOINT [ "python3", "app.py" ]