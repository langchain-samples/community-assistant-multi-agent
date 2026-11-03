"""Per-request user context shared by every agent in the demo."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class UserContext:
    user_id: str
    display_name: str
    roles: tuple[str, ...]
    entitlements: tuple[str, ...]
    purchased_products: tuple[str, ...]
    locale: str = "en_US"

    def __post_init__(self) -> None:
        # Context that comes in as JSON (through the Agent Server API or A2A) has lists.
        # Store tuples so the types match and Pydantic doesn't warn when serializing.
        # A single string is one value, not a list of characters.
        for name in ("roles", "entitlements", "purchased_products"):
            value = getattr(self, name)
            if isinstance(value, str):
                object.__setattr__(self, name, (value,))
            elif not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> UserContext:
        """Build from the JSON form. Raises ``PermissionError`` if the dict is malformed."""
        try:
            return cls(
                user_id=d["user_id"],
                display_name=d["display_name"],
                roles=d.get("roles", ()),
                entitlements=d.get("entitlements", ()),
                purchased_products=d.get("purchased_products", ()),
                locale=d.get("locale", "en_US"),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise PermissionError(f"malformed user context: {exc!r}") from exc


_ADMIN = UserContext(
    user_id="u-1001",
    display_name="Alex Morgan",
    roles=("customer", "can_create_case"),
    entitlements=("HCM", "Payroll"),
    purchased_products=("Acme HCM", "Acme Payroll"),
)

DEMO_USERS: dict[str, UserContext] = {
    "admin_customer": _ADMIN,
    "partner_readonly": UserContext(
        user_id="u-2002",
        display_name="Sam Rivera",
        roles=("partner",),
        entitlements=("HCM",),
        purchased_products=("Acme HCM",),
    ),
    "fr_customer": replace(_ADMIN, user_id="u-3003", display_name="Camille Dubois", locale="fr_FR"),
}


def require_user_context(runtime_context: UserContext | dict | None) -> UserContext:
    """Return a UserContext from either form. Raises ``PermissionError`` if it's missing or malformed."""
    if runtime_context is None:
        raise PermissionError("missing user context")
    if isinstance(runtime_context, UserContext):
        return runtime_context
    return UserContext.from_dict(runtime_context)
