# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Test scaffolding for pose estimation reproducibility testing."""

import json
import os
from typing import TypedDict
import unittest

import matplotlib.pyplot as plt
import numpy as np


RUN_TEST = os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'manual_on_robot'


class Translation(TypedDict):
    x: float
    y: float
    z: float


class Quaternion(TypedDict):
    x: float
    y: float
    z: float
    w: float


class Pose(TypedDict):
    position: Translation
    orientation: Quaternion
    identifier: str
    frame_id: str
    pose_name_for_tf: str


class BackendMetrics(TypedDict):
    x_translations_x_mean_in_meters: float
    x_translations_x_std_in_meters: float
    x_translations_y_mean_in_meters: float
    x_translations_y_std_in_meters: float
    x_translations_z_mean_in_meters: float
    x_translations_z_std_in_meters: float


class PoseEstimationReproducibilityTest(unittest.TestCase):
    """
    Unit tests for Pose estimation repeatability and variance.

    This test will take json files that contain pose estimation results from different
    depth backends and compare the results. The way this test will do this is:

    1. For the same depth backend, plot the mean and standard deviation of the translation.
    2. For the same depth backend, plot the mean and standard deviation of the rotation.

    3. To compare different backends, get the error between the mean of the translation and
       rotation.
    """

    poses: dict[str, list[Pose]]
    _output_dir: str

    @classmethod
    def setUpClass(cls) -> None:
        """Run once before all tests."""
        if not RUN_TEST:
            return
        cls.json_folder_path = os.environ.get('POSE_FOLDER')
        if cls.json_folder_path is None:
            raise ValueError('POSE_FOLDER is not set')

        cls.json_files = [f for f in os.listdir(cls.json_folder_path) if f.endswith('.json')]
        if len(cls.json_files) == 0:
            raise ValueError('No json files found in the POSE_FOLDER')

        cls._output_dir = f'{cls.json_folder_path}/plots'
        os.makedirs(cls._output_dir, exist_ok=True)

        # Now read the json files and store the poses in a list
        cls.poses = {}
        for file in cls.json_files:
            with open(os.path.join(cls.json_folder_path, file), 'r') as f:
                cls.poses[file] = []
                for pose in json.load(f):
                    cls.poses[file].append(Pose(
                        position=Translation(
                            x=pose['position']['x'],
                            y=pose['position']['y'],
                            z=pose['position']['z']
                        ),
                        orientation=Quaternion(
                            x=pose['orientation']['x'],
                            y=pose['orientation']['y'],
                            z=pose['orientation']['z'],
                            w=pose['orientation']['w']
                        ),
                        identifier=file,
                        frame_id=pose['frame_id'],
                        pose_name_for_tf=pose['pose_name_for_tf']
                    ))

    @classmethod
    def tearDownClass(cls) -> None:
        """Run once after all tests."""
        pass

    def setUp(self) -> None:
        """Run before each test."""
        pass

    def tearDown(self) -> None:
        """Run after each test."""
        pass

    def _plot_translation_metric(
        self,
        translations: np.ndarray,
        metric_name: str,
        backend: str
    ) -> dict[str, float]:
        translations_mean_in_meters = np.mean(translations)
        translations_std_in_meters = np.std(translations)
        plt.hist(translations, bins=30, density=True, alpha=0.7)
        plt.axvline(translations_mean_in_meters,
                    color='r',
                    linestyle='--',
                    label=f'mean={translations_mean_in_meters:.3f} meters')
        plt.axvline(translations_mean_in_meters - translations_std_in_meters,
                    color='k',
                    linestyle=':',
                    label=f'std={translations_std_in_meters:.3f}')
        plt.axvline(translations_mean_in_meters + translations_std_in_meters,
                    color='k',
                    linestyle=':')
        plt.legend()
        plt.xlabel(f'{metric_name} (meters)')
        plt.ylabel('Density')
        plt.title(f'{backend} {metric_name}')
        # Save fig to a file
        plt.savefig(f'{self._output_dir}/{backend}_{metric_name}.png')
        plt.close()
        return {
            'mean': translations_mean_in_meters,
            'std': translations_std_in_meters
        }

    def _get_metrics_per_backend(self, poses: list[Pose],
                                 backend: str) -> dict[str, float]:
        """
        Get the metrics for a given backend.

        Args
        ----
            poses: list of poses
            backend: str

        Returns
        -------
            dict[str, float]: Dictionary with the metrics for the given backend
            Saves a csv with the std dev in cm in x,y and z direction.
            Also saves plots for each backend, the std dev in cm in x,y and z direction.

        """
        # Firstly plot the mean, stad dev of just the x component of the translation
        x_translations = np.array([pose['position']['x'] for pose in poses],
                                  dtype=np.float32)
        x_translations_metrics = self._plot_translation_metric(
            x_translations, 'x_translations', backend)

        y_translations = np.array([pose['position']['y'] for pose in poses],
                                  dtype=np.float32)
        y_translations_metrics = self._plot_translation_metric(
            y_translations, 'y_translations', backend)

        z_translations = np.array([pose['position']['z'] for pose in poses],
                                  dtype=np.float32)
        z_translations_metrics = self._plot_translation_metric(
            z_translations, 'z_translations', backend)

        # Save a csv just plotting for each backend, the std dev in cm in x,y and z direction.
        with open(f'{self._output_dir}/{backend}_std_dev_cm.csv', 'w') as f:
            f.write('x_cm,y_cm,z_cm\n')
            f.write(
                f'{x_translations_metrics["std"] * 100},'
                f'{y_translations_metrics["std"] * 100},'
                f'{z_translations_metrics["std"] * 100}\n')

    # ---- Tests ---------------------------------------------------------------
    def test_per_backend_metrics(self) -> None:
        """It behaves correctly on valid input."""
        if not RUN_TEST:
            self.skipTest('RUN_TEST is not set to true')
        for backend in self.poses:
            self._get_metrics_per_backend(self.poses[backend], backend)

    def test_compare_backends(self) -> None:
        """Compare the metrics of the different backends."""
        if not RUN_TEST:
            self.skipTest('RUN_TEST is not set to true')
        # skip this test
        self.skipTest('Not yet implemented')


if __name__ == '__main__':
    unittest.main(verbosity=2)
