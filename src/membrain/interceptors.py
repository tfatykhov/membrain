"""gRPC interceptors for Membrain.

Provides:
- LoggingInterceptor: Request/response logging with timing
"""

import time
import uuid

import grpc

from membrain.logging import get_logger, request_id_var, session_id_var

logger = get_logger(__name__)


class LoggingInterceptor(grpc.ServerInterceptor):
    """gRPC interceptor that logs all RPC calls with timing."""

    def intercept_service(
        self,
        continuation: grpc.RpcMethodHandler,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Intercept and log RPC calls.

        Args:
            continuation: The next handler in the chain.
            handler_call_details: Details about the RPC call.

        Returns:
            The RPC method handler.
        """
        # Extract method name (e.g., "/memory_bridge.MemoryUnit/Remember")
        method = handler_call_details.method.split("/")[-1] if handler_call_details.method else "unknown"

        # Get the original handler
        handler = continuation(handler_call_details)
        if handler is None:
            return None  # type: ignore[return-value]

        # Extract session_id from metadata if provided
        metadata = dict(handler_call_details.invocation_metadata or [])
        session_id = metadata.get("session-id") or metadata.get("x-session-id")

        def wrap_unary_unary(
            original_handler: grpc.RpcMethodHandler,
        ) -> grpc.RpcMethodHandler:
            """Wrap unary-unary handler with logging."""

            def logged_handler(
                request: object,
                context: grpc.ServicerContext,
            ) -> object:
                # Set context variables
                req_id = str(uuid.uuid4())[:8]
                request_id_var.set(req_id)
                if session_id:
                    session_id_var.set(session_id)

                start_time = time.perf_counter()
                success = True
                error_code = None

                try:
                    response = original_handler.unary_unary(request, context)
                    return response
                except Exception as e:
                    success = False
                    error_code = str(type(e).__name__)
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    log_extra = {
                        "method": method,
                        "duration_ms": round(duration_ms, 2),
                        "success": success,
                    }
                    if error_code:
                        log_extra["error_code"] = error_code

                    if success:
                        logger.info("RPC completed", extra=log_extra)
                    else:
                        logger.error("RPC failed", extra=log_extra)

                    # Clear context variables
                    request_id_var.set(None)
                    session_id_var.set(None)

            # Create new handler with wrapped method
            return grpc.unary_unary_rpc_method_handler(
                logged_handler,
                request_deserializer=original_handler.request_deserializer,
                response_serializer=original_handler.response_serializer,
            )

        # Only wrap unary-unary handlers for now
        if handler.unary_unary:
            return wrap_unary_unary(handler)

        return handler
