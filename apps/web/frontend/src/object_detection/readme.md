# Object Detection Frontend Notes

The module owns still-image, camera, and local-video detection views plus their
bounding-box rendering.

Real-time modes keep at most one in-flight request and read the newest available
frame after it completes. They do not accumulate a frame queue. Fixed-rate video
analysis seeks and processes scheduled sample points; maximum-throughput mode is
allowed to skip intermediate frames.

Frames are scaled in the browser before encoding to reduce transport and decode
cost. Overlay coordinates remain based on the source frame dimensions returned
by the API. Stopping, changing mode, leaving the route, or unmounting cancels the
request loop and releases all camera tracks.

Private camera/video components stay in this directory. The common overlay moves
to a shared vision component only when another App consumes the same data
contract.
