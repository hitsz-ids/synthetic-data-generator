Our motivation of this project
========================================

Synthetic data? what for?
---------------------------------------

Generally, we want to generate synthetic data for various reasons: [#1]_ [#2]_ [#3]_ [#4]_

- **Data Privacy**: Synthetic data can be generated without using sensitive customer data such as
  Personally Identifiable Information (PII) or Personal Health Information (PHI).
  This helps to address privacy concerns and comply with privacy regulations such as GDPR and CCPA.
- **Cost and Availability**: Some types of data are costly to collect, or they are rare.
  Synthetic data generation is inexpensive compared to collecting large datasets.
  It can be generated to meet specific needs or conditions that are not available in existing (real) data.
- **AI/ML Model Development**: Synthetic data is being used more frequently for the training of
  machine learning models because of its benefit to data privacy. It's estimated that by 2024,
  60% of the data used to develop AI and analytics projects will be synthetically generated.
- **Flexibility and Control**: Companies can customize synthetic datasets to meet their specific needs,
  including manipulating variables and parameters to generate different scenarios and
  test various hypotheses.
- **Data Acquisition and Cleaning**: Synthetic data relieves you of the burdens of manual data acquisition, annotation, and cleaning.

.. [#1] https://research.aimultiple.com/synthetic-data/
.. [#2] https://indiaai.gov.in/article/synthetic-data-description-benefits-and-implementation
.. [#3] https://www.klippa.com/en/blog/information/what-is-synthetic-data/
.. [#4] https://www.ibm.com/topics/synthetic-data


SDV: SOTA, and not perfact
---------------------------------------------------------

In this case, we found `SDV <https://github.com/sdv-dev/SDV>`_,
a Python library designed to be your one-stop shop for creating tabular synthetic data.
`In this research <https://dai.lids.mit.edu/wp-content/uploads/2018/03/SDV.pdf>`_,
they propose techniques for data synthesis against associative relationships in relational databases and open source it as SDV.

However, while SDV is a powerful tool for generating synthetic data, it is not without its limitations.
One of the main challenges we encountered during our usage was related to performance.
The process of generating synthetic data with SDV can be computationally intensive,
especially when dealing with large and complex datasets.
This can lead to longer processing times and increased demand on system resources,
which might not be feasible for all use cases or environments.

Second, the architecture of SDV presents certain limitations when it comes to extending its capabilities.
While SDV is designed to support few data synthesis techniques for relational databases,
its architecture makes it difficult to incorporate additional algorithms or support different modalities of data.
This restricts our ability to expand upon its functionality and adapt it to a wider range of use cases.

In addition to the performance and architecture issues, another challenge we faced was related to the licensing of SDV.
`In decenber 2022 <https://github.com/sdv-dev/SDV/pull/1150>`_,
SDV has switched to the `Business Source License (BSL) 1.1 <https://github.com/sdv-dev/SDV/blob/main/LICENSE>`_.
While this license allows for free usage of the software, it restricts certain types of modifications and derivative works,
which means that our improvements will become part of their commercial functionality.

What can we do?
-------------------------------------

To address these challenges, we intend to design a new system that is efficient, scalable,
and capable of simulating databases at the scale of tens of millions.
This new system will be designed with a flexible architecture
that can easily incorporate additional algorithms and support different types of data.
Furthermore, it will be licensed under the `Apache 2.0 license <https://github.com/hitsz-ids/synthetic-data-generator/blob/main/LICENSE>`_,
which allows for greater freedom in terms of modifications and derivative works.

By developing this new system, we aim to advance the research and development of synthetic data,
providing a more robust and flexible tool for data scientists and researchers.
This will not only enhance the quality and representativeness of synthetic data but also promote its use
in a wider range of applications,
ultimately contributing to the advancement of data science and machine learning.
We believe that with these improvements,
synthetic data will play an even more crucial role in the future of data-driven decision making and data security.


.. TODO: When we have time, we will release detailed performance comparisons between our new system and SDV.
.. NOTE::

  Absolutely, we understand the importance of providing clear and comprehensive comparisons when introducing a new system. In the near future, we plan to release detailed performance comparisons between our new system and SDV. This will include comparisons of efficiency, scalability, and the ability to handle different data modalities. We will also provide a feature comparison to highlight the improvements and additional capabilities of our new system.

  We believe that this information will be valuable for users to understand the advantages of our system and make informed decisions about its use. Please stay tuned for these updates. We are excited about the potential of our new system and look forward to sharing more details with you soon.


What's next?
-------------------------------------

View our :ref:`architecture` for more details.

Follow our project on `GitHub <https://github.com/hitsz-ids/synthetic-data-generator>`_.

Help us improve our project on `GitHub Issues <https://github.com/hitsz-ids/synthetic-data-generator/issues>`_.

Start contributing: :ref:`developer_guides`.
