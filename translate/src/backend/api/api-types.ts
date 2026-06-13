import type { ApiJsonValue, ApiSuccessEnvelope } from "../../shared/error";

export type { ApiJsonValue, ApiSuccessEnvelope } from "../../shared/error";

export interface ApiErrorEnvelope {
  ok: false;
  error: {
    code: string;
    message: string;
    details?: ApiJsonValue;
  };
}

export type ApiEnvelope = ApiSuccessEnvelope | ApiErrorEnvelope;

export function ok(data: ApiJsonValue): ApiSuccessEnvelope {
  return { ok: true, data };
}
