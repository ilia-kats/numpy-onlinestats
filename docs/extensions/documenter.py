from sphinx.ext.autodoc import AttributeDocumenter


class NanobindMethodDocumenter(AttributeDocumenter):
    objtype = "attribute"

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return member.__class__.__name__ != "nb_method" and super().can_document_member(
            member, membername, isattr, parent
        )


def setup(app):
    app.add_autodocumenter(NanobindMethodDocumenter)
