# 1단계: 빌드
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

# 프로덕션 빌드 (API는 Gateway로 직접 요청)
# VITE_API_BASE를 빈 문자열로 설정하면 nginx가 /api → Gateway로 프록시
ARG VITE_API_BASE=/api
ENV VITE_API_BASE=${VITE_API_BASE}

RUN npm run build

# 2단계: nginx 서빙
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
