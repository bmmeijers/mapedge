from mapedge.lib.ransac import calculate_ransac_iterations


def fit(J):
    print("hi from fit")
    max_iterations = calculate_ransac_iterations(0.5, 0.99999)
    threshold_pixel_dist = 5
    sample_point_count = 2
    threshold_count_confidence = 8
    # process_sheet(
    #     J,
    #     max_iterations,
    #     threshold_pixel_dist,
    #     sample_point_count,
    #     threshold_count_confidence,
    # )
