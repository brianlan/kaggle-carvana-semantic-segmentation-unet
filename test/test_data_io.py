from data_io import ImageFileName


def test_image_file_name():
    i = ImageFileName('test')
    assert i.jpg == 'test.jpg'
    assert i.png == 'test.png'
