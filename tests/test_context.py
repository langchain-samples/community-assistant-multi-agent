import pytest

from community_assistant.context import DEMO_USERS, UserContext, require_user_context
from community_assistant.localization import failure_message


def test_require_user_context_raises_without_context():
    with pytest.raises(PermissionError, match="missing user context"):
        require_user_context(None)


def test_require_user_context_accepts_dict():
    user = DEMO_USERS["admin_customer"]
    assert require_user_context(user.to_dict()) == user
    assert require_user_context(user) is user
    assert UserContext.from_dict(user.to_dict()) == user


def test_failure_message_falls_back_to_en_us():
    assert failure_message("de_DE") == failure_message("en_US")


def test_list_fields_are_stored_as_tuples():
    user = UserContext(
        user_id="u", display_name="U", roles=["customer"], entitlements=["HCM"],
        purchased_products=["Acme HCM"],
    )
    assert user.roles == ("customer",) and user.entitlements == ("HCM",)
    assert user.purchased_products == ("Acme HCM",)
    # A lone string is one value, never split into characters.
    assert UserContext("u", "U", "customer", (), ()).roles == ("customer",)


@pytest.mark.parametrize(
    "bad",
    [{}, {"user_id": "u"}, {"user_id": "u", "display_name": "U", "roles": 5}, "admin", 42],
    ids=["empty", "missing-name", "bad-roles", "string", "int"],
)
def test_malformed_context_fails_closed_with_permission_error(bad):
    with pytest.raises(PermissionError):
        require_user_context(bad)
