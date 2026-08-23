# Yolo Fruit Slice Backend

## Summary

This backend module registers the Yolo Fruit Slice game and exposes a lightweight discovery/status endpoint. The game simulation and camera loop are frontend-owned. Body pose inference is reused through the platform Pose Estimation API; this module does not acquire a model or maintain a game session.

The manifest requires model `ultralytics/yolov8-pose` with capability `pose-estimation` so the App registry can report whether the game dependency is available.

```mermaid
flowchart LR
    Registry[App Registry] --> Manifest[Game manifest]
    Browser --> Status[Game status route]
    Status --> Browser
    Camera --> Frontend[Frontend game loop]
    Frontend --> PoseAPI[/api/v1/apps/pose-estimation/estimate/frame]
    PoseAPI --> Frontend
```

## Module structure

| File | Responsibility |
| --- | --- |
| `manifest.py` | Game identity, frontend/API routes, and pose dependency. |
| `blueprint.py` | Exposes the manifest and status router to the platform. |
| `routes.py` | Returns the readiness marker and canonical inference route. |

There is intentionally no backend service, schema package, physics engine, score store, or model adapter in this directory.

## Interfaces

API prefix: `/api/v1/games/yolo-fruit-slice`

| Method | Route | Contract |
| --- | --- | --- |
| GET | `/api/v1/games/yolo-fruit-slice/status` | Return `status=ready` and `/api/v1/apps/pose-estimation/estimate/frame` as the inference endpoint. |

The status response means the backend route is available. Model installation and loadability are represented separately by the App registry and model APIs.

## Runtime boundary

The frontend turns pose keypoints into forearm blades, advances fruit and bomb motion, detects intersections, and owns score/combo state. It calls the shared Pose Estimation frame endpoint for inference. Keeping this split avoids a second YOLO model lifecycle and prevents a browser-local animation clock from being duplicated on the server.

The status route is stateless, performs no I/O beyond response construction, and has no cancellation or resource-release work.

## Errors, observability, and security

The endpoint accepts no body, query string, model path, or user-controlled storage identifier. Platform middleware owns trace IDs and access logging. Model and camera failures occur through their respective platform or browser interfaces, not this route.

## Tests and review points

Tests should verify blueprint registration, manifest model/capability facts, the API prefix, response envelope, readiness value, and canonical inference path.

Review changes for model acquisition or game state added to this module, a duplicated inference endpoint, status implying that resources are installed, or server-side frame storage.
