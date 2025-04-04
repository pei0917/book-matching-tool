from v1 import match_books

image_path = "examples/example_1.jpg"
reference_images = {
    "book1": "examples/1.jpg",
    "book2": "examples/2.jpg",
    "book3": "examples/3.jpg",
    "book4": "examples/4.jpg",
    "book5": "examples/5.jpg",
    "book6": "examples/6.jpg",
}
print(match_books(image_path, reference_images))