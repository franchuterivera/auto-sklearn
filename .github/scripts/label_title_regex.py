from ghapi.all import context_github
from ghapi.all import GhApi
from ghapi.all import user_repo
from ghapi.all import github_token
import re

owner, repo = user_repo()
issues = context_github.event.issues
title = issues.title

regex_to_labels = [
    (r"\bDOC\b", "Documentation"),
    (r"\bBUG\b", "Bug")
]

labels_to_add = [
    label for regex, label in regex_to_labels
    if re.search(regex, title)
]

if labels_to_add:
    api = GhApi(owner=owner, repo=repo, token=github_token())
    api.issues.add_labels(issues.number, labels=labels_to_add)
