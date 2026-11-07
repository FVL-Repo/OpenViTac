"""The ``openvitac`` application object and its command registrations."""

import cyclopts

# Commands load lazily so ``sample-seeds`` never imports ``openvitac_sim``
# (and through it the app launch machinery).  Detailed help comes from each
# command's docstring when it is resolved.
main = cyclopts.App()
main.command("openvitac_cli.commands.collect:collect")
main.command("openvitac_cli.commands.sample_seeds:sample_seeds")
main.command("openvitac_cli.commands.tasks:tasks")
