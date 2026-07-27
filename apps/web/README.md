# hugging-mac Web

Web 层采用彻底分离的两个工程：

- `backend/`：FastAPI、平台 API、App Blueprint 与模型业务编排；
- `frontend/`：Vue + Vite、平台首页、模型页和各 App 页面。

开发时分别启动：

```bash
uv run hugging-mac-web
```

```bash
cd apps/web/frontend
npm install
npm run dev
```

Vue dev server 会将 `/api` 代理到 `http://127.0.0.1:8000`。
