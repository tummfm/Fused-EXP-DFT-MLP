.. _documentation_guide:

Documentation Styleguide
========================

Files
------

- Documentation root::

	mydocs/
	
- Documentation source::

	mydocs/source/
	
- Documentation build::

	mydocs/build/

Building the Documentation
---------------------------

1. Install sphinx::

	pip install sphinx

2. Navigate to documentation directory::

	cd mydocs
	make html

3. Host the website, e. g. by::

	python -m http.server --directory mydocs/build/html


Docstring Style
----------------

From <https://google.github.io/styleguide/pyguide.html#Comments:>


- Functions and Methods::

	"""Short description.
	
		A longer description if necessary
	
		Args:
			name: value
			other_var:
				A blank line is allowed
		
		Returns:
			Description of return type.
			With example::
		
				{"some": "dict", "with": ("a", "tuple")}
	
	
		Raises:
			SomeError: With a description.
	
	"""

- Classes::

	"""Class summary.
	
		Longer information...
		Longer information...
	
	
		Attributes:
			one_attribute_name: description of value
			another_attribute_name: description of value
	
	
	""""

- Files::

	"""A one line summary of the file.
	
		There should be a description of the content of the file, optionally with use cases.
	
		For example::
		
			SomeObject = SommeClass(some_variable)
	
	"""

Guides
-----------

Usefull guides: 


- **Google Python Styleguide:** <https://google.github.io/styleguide/pyguide.html>
- **Markdown Guide** <https://www.markdownguide.org/basic-syntax/>
- **Sphinx** < https://www.sphinx-doc.org/en/master/>