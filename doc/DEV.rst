Development
===========

This chapter targets developers who want to contribute to the project's core.

.. mermaid::

   graph TD
       subgraph _core
           base_modules
           core_implementations
           registry
       end

       subgraph runners
           SlurmRunner
           StandaloneRunner
       end

       subgraph installers
           SlurmInstaller
           StandaloneInstaller
       end

       subgraph systems
           SlurmSystem
           StandaloneSystem
       end

       installers --> _core
       runners --> _core
       systems --> _core

Core Modules
------------

We use `import-linter <https://github.com/seddonym/import-linter>`_ to ensure no core modules import higher level modules.

``Registry`` object is a singleton that holds implementation mappings. Users can register their own implementations to the registry or replace the default implementations.

Optional agent dependencies
---------------------------

If an agent needs an optional package, register an ``UnavailableAgent``
subclass when the package is missing. Other tests in the same directory can
then be parsed.

.. code-block:: python

   from cloudai.core import Registry, UnavailableAgent

   try:
       import optional_agent_package
   except ImportError:
       class MissingOptionalAgent(UnavailableAgent):
           reason = "Install the 'optional-agent-package' package to use this agent."

       agent_class = MissingOptionalAgent
   else:
       agent_class = optional_agent_package.OptionalAgent

   Registry().add_agent("optional_agent", agent_class)

The placeholder validates ``BaseAgentConfig`` settings and accepts other fields
without checking them. Override ``get_config_class()`` to return the real
configuration model if it can be imported without the optional package.

Parsing and ``verify-configs`` log a warning for unavailable agents. ``run`` and
``dry-run`` fail before installation or submission if a selected DSE test needs
one, including hooks and single-sbatch runs. Unknown agent names remain errors.
Using the placeholder directly raises ``ImportError`` with its configured reason.

Cache
-----

Some prerequisites can be installed. For example:

Docker images, git repos with executable scripts, etc. All such "installables" are kept under the system's ``install_path``.

Installables are shared among all tests. Therefore, if any number of tests use the same installable, it is installed only once for a particular system TOML.

.. mermaid::

   classDiagram
       class Installable {
           <<abstract>>
           + __eq__(other: object)
           + __hash__()
       }

       class DockerImage {
           + url: str
           + install_path: str | Path
       }

       class GitRepo {
           + git_url: str
           + commit_hash: str
           + install_path: Path
       }

       class PythonExecutable {
           + git_repo: GitRepo
           + venv_path: Path
       }

       Installable <|-- DockerImage
       Installable <|-- GitRepo
       Installable <|-- PythonExecutable
       PythonExecutable --> GitRepo

       class BaseInstaller {
           <<abstract>>
           + install(items: Iterable[Installable])
           + uninstall(items: Iterable[Installable])
           + is_installed(items: Iterable[Installable]) -> bool

           * install_one(item: Installable)
           * uninstall_one(item: Installable)
           * is_installed_one(item: Installable) -> bool
       }

       BaseInstaller <|-- SlurmInstaller
       BaseInstaller <|-- StandaloneInstaller
