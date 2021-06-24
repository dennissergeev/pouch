"""Working with Path-like objects."""
from pathlib import Path


def full_path_glob(path):
    """Recursive glob, including directories with regex."""
    resolved = Path(path.absolute().parts[0])
    glob_parts = []
    to_path = True
    for part in path.parts[1:]:
        if to_path and ("*" in part):
            to_path = False
        if to_path:
            resolved /= part
        else:
            glob_parts.append(part)
    gen = resolved.rglob(path.anchor.join(glob_parts))
    return gen


def lsdir(path):
    """List all files and directories in the given path, sorted."""
    return sorted(path.iterdir())


class DisplayablePath:
    """
    Tree-like representation of a path.

    Usage:
    ------
    >>> paths = DisplayablePath.make_tree(Path("foo"))
    >>> for path in paths:
            print(path.displayable())

    """

    display_filename_prefix_middle = "├──"
    display_filename_prefix_last = "└──"
    display_parent_prefix_middle = "    "
    display_parent_prefix_last = "│   "

    def __init__(self, path, parent_path, is_last):
        """Initialise DisplayablePath."""
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        """Make a path tree from the given top-level path."""
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(
            (path for path in root.iterdir() if criteria(path)),
            key=lambda s: str(s).lower(),
        )
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(
                    path, parent=displayable_root, is_last=is_last, criteria=criteria
                )
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):  # noqa
        if self.path.is_dir():
            return self.path.name + "/"
        return self.path.name

    def displayable(self):  # noqa
        if self.parent is None:
            return self.displayname

        _filename_prefix = (
            self.display_filename_prefix_last
            if self.is_last
            else self.display_filename_prefix_middle
        )

        parts = ["{!s} {!s}".format(_filename_prefix, self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(
                self.display_parent_prefix_middle
                if parent.is_last
                else self.display_parent_prefix_last
            )
            parent = parent.parent

        return "".join(reversed(parts))
