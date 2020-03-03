keras_biological_gaps_sequences
=========================================================================================
|travis| |sonar_quality| |sonar_maintainability| |codacy|
|code_climate_maintainability| |pip| |downloads|

Python package to generate on-hot encoded biological gaps to use for training and prediction.

How do I install this package?
----------------------------------------------
As usual, just download it using pip:

.. code:: shell

    pip install keras_biological_gaps_sequences

Tests Coverage
----------------------------------------------
Since some software handling coverages sometimes
get slightly different results, here's three of them:

|coveralls| |sonar_coverage| |code_climate_coverage|

Available datasets
-----------------------------------------------
Currently, there is only a dataset of gaps available
within the package: the mapping of known gaps from hg19
to hg38. In the future, we will be adding more mapping.

Usage example
-----------------------------------------------
To use the sequence you can do as follows:

.. code:: Python

    biological_gap_sequence = BiologicalGapsSequence(
        source="hg19",
        target="hg38",
        source_window_size=1000,
        target_window_size=1000,
        batch_size=32
    )

    model = build_my_denoiser()
    model.fit_generator(
        biological_gap_sequence,
        steps_per_epoch=biological_gap_sequence.steps_per_epoch,
        epochs=2,
        shuffle=True
    )

.. |travis| image:: https://travis-ci.org/LucaCappelletti94/keras_biological_gaps_sequences.png
   :target: https://travis-ci.org/LucaCappelletti94/keras_biological_gaps_sequences
   :alt: Travis CI build

.. |sonar_quality| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_keras_biological_gaps_sequences&metric=alert_status
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_keras_biological_gaps_sequences
    :alt: SonarCloud Quality

.. |sonar_maintainability| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_keras_biological_gaps_sequences&metric=sqale_rating
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_keras_biological_gaps_sequences
    :alt: SonarCloud Maintainability

.. |sonar_coverage| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_keras_biological_gaps_sequences&metric=coverage
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_keras_biological_gaps_sequences
    :alt: SonarCloud Coverage

.. |coveralls| image:: https://coveralls.io/repos/github/LucaCappelletti94/keras_biological_gaps_sequences/badge.svg?branch=master
    :target: https://coveralls.io/github/LucaCappelletti94/keras_biological_gaps_sequences?branch=master
    :alt: Coveralls Coverage

.. |pip| image:: https://badge.fury.io/py/keras-biological-gaps-sequence.svg
    :target: https://badge.fury.io/py/keras-biological-gaps-sequence
    :alt: Pypi project

.. |downloads| image:: https://pepy.tech/badge/keras-biological-gaps-sequence
    :target: https://pepy.tech/badge/keras-biological-gaps-sequence
    :alt: Pypi total project downloads

.. |codacy| image:: https://api.codacy.com/project/badge/Grade/90f25e6d3ab3448d9da0401f441dff79
    :target: https://www.codacy.com/manual/LucaCappelletti94/keras_biological_gaps_sequences?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=LucaCappelletti94/keras_biological_gaps_sequences&amp;utm_campaign=Badge_Grade
    :alt: Codacy Maintainability

.. |code_climate_maintainability| image:: https://api.codeclimate.com/v1/badges/0bc73c94073503d4d54a/maintainability
    :target: https://codeclimate.com/github/LucaCappelletti94/keras_biological_gaps_sequences/maintainability
    :alt: Maintainability

.. |code_climate_coverage| image:: https://api.codeclimate.com/v1/badges/0bc73c94073503d4d54a/test_coverage
    :target: https://codeclimate.com/github/LucaCappelletti94/keras_biological_gaps_sequences/test_coverage
    :alt: Code Climate Coverate
