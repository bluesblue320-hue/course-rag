"""Storage backend package: protocols, runtimes, and the pgvector store.

Importing this package never connects to a database.  Modules are imported
explicitly by their consumers so the package has no eager cross-imports.
"""
