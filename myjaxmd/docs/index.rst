Welcome to Chemtrain's documentation!
=====================================

Chemtrain is a collection of training algorithms to parametrize MD potentials.
It builds on MD functionality provided by JAX MD and occasionally extends them.
These extensions are gathered in the jax_md_mod module.

Subheadings
----------------------------

Subsubheading
+++++++++++++++++++++++++++++

This *documentation* ist still in its **infancey**!
For documentation building, possibly start with jax_sgmc.

Possibly good api: Have dedicated file for trainers as first landing point for
users, similar to alias.py and create chickstart page. All secondary functions
go into other files, where onkly API needs to be documented.



.. warning::
   * Refactor code to adhere to Google style guide <https://google.github.io/styleguide/pyguide.html> `
   * Run pylint to check code style
   * Add type checking, if possible

.. admonition:: Hint!

   You can check the documentation template of classes at https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html



.. admonition:: TODO:

   Once the api has settled down:
   * Write for each file a dedicated .rst file clustering functions together
   * Each File should include its own documentation (file heading) including
   doctests and examples!

.. toctree::
   :maxdepth: 1
   :caption: Quickstart





.. toctree::
   :maxdepth: 2
   :caption: JAX MD extensions

   chemtrain.jax_md_mod.custom_nn



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
