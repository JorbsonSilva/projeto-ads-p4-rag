import os
from dataclasses import dataclass, field

import jwt


@dataclass(frozen=True)
class InternalAuth:
    secret: str = field(repr=False)
    issuer: str
    audience: str

    @classmethod
    def from_env(cls):
        secret = os.environ.get("RAG_INTERNAL_JWT_SECRET", "")
        issuer = os.environ.get("RAG_INTERNAL_JWT_ISSUER", "orienta-spring")
        audience = os.environ.get("RAG_INTERNAL_JWT_AUDIENCE", "orienta-rag")
        if len(secret.encode()) < 32 or not issuer.strip() or not audience.strip():
            raise ValueError("Configure o segredo JWT interno (mínimo de 32 bytes), emissor e audiência.")
        return cls(secret, issuer, audience)

    def subject(self, token):
        claims = jwt.decode(
            token, self.secret, algorithms=["HS256"], issuer=self.issuer,
            audience=self.audience, leeway=0,
            options={"require": ["iss", "aud", "sub", "iat", "exp"], "strict_aud": True},
        )
        if (
            type(claims["iat"]) is not int or type(claims["exp"]) is not int
            or not 0 < claims["exp"] - claims["iat"] <= 120
            or not claims["sub"].strip() or len(claims["sub"]) > 200
        ):
            raise jwt.InvalidTokenError("Credencial interna inválida.")
        scope = claims.get("scope")
        if not isinstance(scope, str) or "rag:internal" not in scope.split():
            raise PermissionError("Escopo insuficiente.")
        return claims["sub"]
