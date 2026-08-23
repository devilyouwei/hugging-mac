# Web Frontend Technical Overview

The frontend is one Vue 3, TypeScript, and Vite application. It owns the platform
shell, navigation, Model Library, App/Game routes, browser media, and API clients.

## Routing and modules

The root router explicitly registers platform and App pages. Each App directory
owns its page, API calls, types, composables, and private components. App modules
cannot import one another. Shared components and utilities live at the frontend
root only when they implement a stable cross-App contract.

The backend catalog supplies App/model presentation data and availability. It
does not select Vue components dynamically; the frontend keeps that mapping in
trusted source code.

## API boundary

Platform clients under `src/api` own catalog, model, resource, instance, media,
and system contracts. An App-specific client stays in the App directory. UI code
never imports Python or the SDK and never constructs local model paths.

Requests expose explicit loading, cancellation, retry, and error state. After a
resource or instance mutation, the relevant catalog snapshot is refreshed rather
than patched with private backend assumptions.

## Layout and responsive behavior

Global tokens and shell styles live under `src/styles`. Pages use a desktop,
tablet, and phone layout with readable typography at each breakpoint. Large
desktop information grids collapse to fewer columns on tablets and to a linear
flow on phones; touch targets remain usable without hover.

App-only overlays, visualizers, and controls remain in the App module. A
component moves into the shared component library only after its data contract is
independent of the originating App.

## Browser media

Camera and microphone tracks are acquired only after user action and are released
on stop, route leave, component unmount, or failure. Long-running workflows own
one cancellation boundary and must not retain unbounded frame/audio queues.

Frontend tests cover public rendering states, API mapping, responsive behavior,
and media cleanup without requiring a real model backend.
