import pytest

from t_bot.errors import BotError
from t_bot.filesystem import FileSystem


@pytest.fixture
def filesystem(tmp_path):
    return FileSystem(tmp_path / "workspace", 262144, (tmp_path / "private",))


def test_file_lifecycle_and_unicode_pagination(filesystem):
    filesystem.create_directory("notes/nested")
    written = filesystem.write_file("notes/nested/hello.txt", "Привіт 😀 світе")
    assert written["written_bytes"] == len("Привіт 😀 світе".encode())
    result = filesystem.read_file("notes/nested/hello.txt", offset=7, limit=1)
    assert result["content"] == "😀"
    assert result["truncated"]
    filesystem.write_file("notes/nested/hello.txt", "updated")
    assert filesystem.read_file("notes/nested/hello.txt")["content"] == "updated"
    listing = filesystem.list_files("notes/nested")
    assert listing["entries"][0]["name"] == "hello.txt"
    filesystem.delete_file("notes/nested/hello.txt")
    assert filesystem.list_files("notes/nested")["entries"] == []


def test_absolute_path_inside_root_is_supported(filesystem):
    path = str(filesystem.root / "hello.txt")
    filesystem.write_file(path, "hello")
    assert filesystem.read_file(path)["content"] == "hello"


@pytest.mark.parametrize(
    "path", ["../escape", "/etc/passwd", ".env", ".env.local", ".envrc", ".git/config"]
)
def test_paths_outside_root_and_secrets_are_rejected(filesystem, path):
    with pytest.raises(BotError):
        filesystem.read_file(path)
    with pytest.raises(BotError):
        filesystem.write_file(path, "bad")


def test_symlinks_are_not_followed_for_read_write_or_delete(filesystem, tmp_path):
    filesystem.create_directory(".")
    outside = tmp_path / "private"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("secret")
    (filesystem.root / "link").symlink_to(outside, target_is_directory=True)
    (filesystem.root / "file_link").symlink_to(secret)
    for path in ("link/secret.txt", "file_link"):
        with pytest.raises((BotError, OSError)):
            filesystem.read_file(path)
        with pytest.raises((BotError, OSError)):
            filesystem.write_file(path, "overwrite")
    with pytest.raises(BotError):
        filesystem.delete_file("file_link")
    assert secret.read_text() == "secret"


def test_internal_files_and_nonregular_files_are_rejected(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    (private / "state.sqlite3").write_text("internal")
    filesystem = FileSystem(tmp_path, 64, (private,))
    with pytest.raises(BotError):
        filesystem.read_file("private/state.sqlite3")
    assert filesystem.list_files()["entries"] == []
    with pytest.raises(BotError):
        filesystem.delete_file(".")
    with pytest.raises(BotError):
        filesystem.read_file(".")


def test_file_size_encoding_and_listing_limits(filesystem):
    filesystem.max_file_bytes = 8
    with pytest.raises(BotError, match="FILESYSTEM_MAX"):
        filesystem.write_file("large.txt", "x" * 9)
    filesystem.create_directory(".")
    (filesystem.root / "large.txt").write_text("x" * 9)
    with pytest.raises(BotError, match="FILESYSTEM_MAX"):
        filesystem.read_file("large.txt")
    (filesystem.root / "binary").write_bytes(b"\xff")
    with pytest.raises(BotError, match="UTF-8"):
        filesystem.read_file("binary")
    for index in range(105):
        (filesystem.root / f"file{index}").touch()
    listing = filesystem.list_files()
    assert listing["truncated"]
    assert len(listing["entries"]) == 100


def test_protected_paths_cannot_be_accessed_using_different_case(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    (private / "state.sqlite3").write_text("internal")
    filesystem = FileSystem(tmp_path, 64, (private,))
    with pytest.raises(BotError, match="внутрішніх"):
        filesystem.read_file("PRIVATE/state.sqlite3")
