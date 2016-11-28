import unittest
import numpy as np

from wsdt import wsDtSegmentation
from numpy_allocation_tracking.decorators import assert_mem_usage_factor

class TestWsDtSegmentation(unittest.TestCase):

    def _gen_input_data(self, ndim):
        assert ndim in (2,3)

        # Create a volume with 8 sections
        pmap = np.zeros( (101,)*ndim, dtype=np.float32 )

        # Three Z-planes
        pmap[  :2,  :] = 1
        pmap[49:51, :] = 1
        pmap[-2:,   :] = 1

        # Three Y-planes
        pmap[:,   :2] = 1
        pmap[:, 49:51] = 1
        pmap[:, -2:] = 1

        if ndim == 3:
            # Three X-planes
            pmap[:, :, :2] = 1
            pmap[:, :, 49:51] = 1
            pmap[:, :, -2:] = 1
        
        return pmap
        

    def test_simple_3D(self):
        pmap = self._gen_input_data(3)

        debug_results = {}
        ws_output = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, groupSeeds=False, out_debug_image_dict=debug_results)
        seeds = debug_results['seeds'][:]
        assert seeds.max() == 8
        assert ws_output.max() == 8

        # Expect seeds at (25,25,25), (25,25,75), (25,75,25), etc...
        expected_seed_coords = list(np.ndindex((2,2,2)))
        expected_seed_coords = 50*np.array(expected_seed_coords) + 25

        #print "EXPECTED:\n", expected_seed_coords
        #print "SEEDS:\n", np.array(np.where(seeds)).transpose()

        for seed_coord in expected_seed_coords:
            assert seeds[tuple(seed_coord)], "No seed at: {}".format( seed_coord )

    def test_simple_2D(self):
        pmap = self._gen_input_data(2)

        debug_results = {}
        ws_output = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, groupSeeds=False, out_debug_image_dict=debug_results)
        seeds = debug_results['seeds'][:]
        assert seeds.max() == 4
        assert ws_output.max() == 4

        # Expect seeds at (25,25,25), (25,25,75), (25,75,25), etc...
        expected_seed_coords = list(np.ndindex((2,2)))
        expected_seed_coords = 50*np.array(expected_seed_coords) + 25

        #print "EXPECTED:\n", expected_seed_coords
        #print "SEEDS:\n", np.array(np.where(seeds)).transpose()

        for seed_coord in expected_seed_coords:
            assert seeds[tuple(seed_coord)], "No seed at: {}".format( seed_coord )


    def test_border_seeds(self):
        """
        check if seeds at the borders are generated
        """

        # create a volume with membrane evidence everywhere
        pmap = np.ones((50, 50, 50))

        # create funnel without membrane evidence growing larger towards the block border.
        pmap[0, 12:39, 12:39] = 0
        pmap[1:50, 13:38, 13:38] = 0

        debug_results = {}
        _ws_output = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, groupSeeds=False, out_debug_image_dict=debug_results)
        seeds = debug_results['seeds'][:]
        assert seeds.sum() == 1
        assert seeds[0, 25, 25] == 1

    def test_memory_usage(self):
        pmap = self._gen_input_data(3)

        # Wrap the segmentation function in this decorator, to verify it's memory usage.
        ws_output = assert_mem_usage_factor(2.5)(wsDtSegmentation)(pmap, 0.5, 0, 10, 0.1, 0.1, groupSeeds=False)
        assert ws_output.max() == 8

        # Now try again, with groupSeeds=True
        # Note: This is a best-case scenario for memory usage, since the memory 
        #       usage of the seed-grouping function depends on the NUMBER of seeds,
        #       and we have very few seeds in this test.
        ws_output = assert_mem_usage_factor(3.5)(wsDtSegmentation)(pmap, 0.5, 0, 10, 0.1, 0.1, groupSeeds=True)
        assert ws_output.max() == 8

    def test_debug_output(self):
        """
        Just exercise the API for debug images, even though we're not
        really checking the *contents* of the images in this test.
        """
        pmap = self._gen_input_data(3)
        debug_images = {}
        ws_output = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, groupSeeds=False, out_debug_image_dict=debug_images)
        assert ws_output.max() == 8
        
        assert 'thresholded membranes' in debug_images
        assert debug_images['thresholded membranes'].shape == ws_output.shape

        assert 'filtered membranes' in debug_images
        assert debug_images['filtered membranes'].shape == ws_output.shape

    def test_group_close_seeds(self):
        """
        In this test we'll use input data that looks roughly like the following:
        
            0                101               202               303
          0 +-----------------+-----------------+-----------------+
            |        |        |        |        |                 |
            |                 |                 |                 |
            |                 |                 |                 |
            |                                   |                 |
         50 |      x   x             y   y      |        z        |
            |                                   |                 |
            |                 |                 |                 |
            |                 |                 |                 |
            |        |        |        |        |                 |
        101 +-----------------+-----------------+-----------------+

        The x and y markers indicate where seeds will end up.
        
        With groupSeeds=False, we get 5 seed points and 5 final segments.
        But with groupSeeds=True, the two x points end up in the same segment,
        and the two y points end up in the same segment.
        The lone z point will not be merged with anything.
        """
        
        input_data = np.zeros((101, 303), dtype=np.float32)
        
        # Add borders
        input_data[0] = 1
        input_data[-1] = 1
        input_data[:, 0] = 1
        input_data[:, -1] = 1

        # Add complete divider for the z compartment
        input_data[:, 202] = 1

        # Add notches extending from the upper/lower borders
        input_data[  1:10,  51] = 1
        input_data[  1:40, 101] = 1
        input_data[  1:10, 151] = 1
        input_data[-10:-1,  51] = 1
        input_data[-40:-1, 101] = 1
        input_data[-10:-1, 151] = 1
        
        # First, try without groupSeeds
        debug_results = {}
        ws_output = wsDtSegmentation(input_data, 0.5, 0, 0, 0.0, 0.0, groupSeeds=False, out_debug_image_dict=debug_results)
        assert ws_output.max() == 5

        # Now, with groupSeeds=True, the left-hand seeds should 
        # be merged and the right-hand seeds should be merged
        debug_results = {}
        ws_output = wsDtSegmentation(input_data, 0.5, 0, 0, 0.0, 0.0, groupSeeds=True, out_debug_image_dict=debug_results)
        assert ws_output.max() == 3

        assert (ws_output[:,   0: 90] == ws_output[51,  51]).all()
        assert (ws_output[:, 110:190] == ws_output[51, 151]).all()
        assert (ws_output[:, 210:290] == ws_output[51, 251]).all()
        
        # The segment values are different
        assert ws_output[51,51] != ws_output[51, 151] != ws_output[51, 251]

    def test_group_seeds_ram_usage(self):
        """
        The original implementation of the groupSeeds option needed
        a lot of RAM, scaling with the number of seeds by N**2.
        The new implementation does the work in batches, so it
        doesn't need as much RAM.  
        
        Here we create a test image that will result in lots of seeds,
        and we'll verify that RAM usage stays under control.
        
        The test image looks roughly like this (seeds marked with 'x'):
        
        +-----------------------------------------------------+
        |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
        |                                                     |
        |                                                     |
        |       x  x  x  x  x  x  x  x  x  x  x  x  x  x      |
        |                                                     |
        |                                                     |
        +-----------------------------------------------------+
        """
        input_data = np.zeros((101, 20001), dtype=np.float32)

        # Add borders
        input_data[0] = 1
        input_data[-1] = 1
        input_data[:, 0] = 1
        input_data[:, -1] = 1

        # Add tick marks
        input_data[:10, ::10] = 1
        
        # Sanity check, try without groupSeeds, make sure we've got a lot of segments
        ws_output = wsDtSegmentation(input_data, 0.5, 0, 0, 0.0, 0.0, groupSeeds=False)
        assert ws_output.max() > 1900

        # Now check RAM with groupSeeds=True
        ws_output = assert_mem_usage_factor(3.0)(wsDtSegmentation)(input_data, 0.5, 0, 0, 2.0, 0.0, groupSeeds=True)
        assert ws_output.max() == 1        

    def test_out_param(self):
        pmap = self._gen_input_data(2)

        debug_results = {}
        preallocated = np.random.randint( 0, 100, pmap.shape ).astype(np.uint32)
        ws_output = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, groupSeeds=False, out_debug_image_dict=debug_results, out=preallocated)
        assert ws_output is preallocated
        seeds = debug_results['seeds'][:]
        assert seeds.max() == 4
        assert ws_output.max() == 4

        # Also with groupSeeds=True
        preallocated = np.random.randint( 0, 100, pmap.shape ).astype(np.uint32)
        ws_output = wsDtSegmentation(pmap, 0.5, 0, 10, 0.1, 0.1, groupSeeds=True, out_debug_image_dict=debug_results, out=preallocated)
        assert ws_output is preallocated
        assert seeds.max() == 4
        assert ws_output.max() == 4


if __name__ == "__main__":
    import sys
    import logging
    mem_logger = logging.getLogger('numpy_allocation_tracking')
    mem_logger.setLevel(logging.DEBUG)
    mem_logger.addHandler(logging.StreamHandler(sys.stdout))
    unittest.main()
