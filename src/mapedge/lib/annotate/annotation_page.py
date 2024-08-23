# annotation_page.py


def make_annotation_feature(px, py, wx, wy):
    """
    Ties Pixel coordinate to World coordinate

    Pixel coordinate system is top left = (0,0), y down = +, x right = +
    World coordinate system is WGS'84 (unspecified realization)
    """

    # FIXME:
    # the spec speaks about 'resourceCoords' instead of 'pixelCoords'
    # https://iiif.io/api/extension/georef/#35-the-resourcecoords-property
    # but allmaps parser does not like that

    return {
        "type": "Feature",
        "properties": {"pixelCoords": [int(px), int(py)]},
        "geometry": {"type": "Point", "coordinates": [wx, wy]},  # lat  # lon
    }


def make_annotation_item(uuid, iiif_end_point, iiif_full_url, svg_mask, features):
    """Return a Georeference Annotation (JSON encoded) document

    <https://iiif.io/api/extension/georef/>
    """
    return {
        "id": f"{uuid}",
        "type": "Annotation",
        "@context": [
            "http://www.w3.org/ns/anno.jsonld",
            "http://geojson.org/geojson-ld/geojson-context.jsonld",
            "http://iiif.io/api/presentation/3/context.json",
        ],
        "motivation": "georeferencing",
        "target": {
            "type": "Image",
            "source": f"{iiif_full_url}",
            "service": [{"@id": f"{iiif_end_point}", "type": "ImageService2"}],
            "selector": {"type": "SvgSelector", "value": f"{svg_mask}"},
        },
        "body": {
            "type": "FeatureCollection",
            "purpose": "gcp-georeferencing",
            "transformation": {
                # "type": "polynomial",
                # "order": 0
                "type": "thinPlateSpline"
            },
            "features": features,
        },
    }


def make_page(items):
    return {
        "type": "AnnotationPage",
        "@context": ["http://www.w3.org/ns/anno.jsonld"],
        "items": items,
    }
