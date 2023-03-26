import asyncio
from imagesearch import add, search
from PIL import Image


def open_image(directory, file_name):
    """
    Open an image file from a directory.
    :param file_name: The name of the image file to open.
    :param directory: The directory where the image file is located.
    :return: The opened image file, or None if the file could not be opened.
    """
    try:
        file_path = f"{directory}/{file_name}"
        image = Image.open(file_path)
        return image
    except Exception as e:
        print(f"Unable to open image file '{file_path}': {str(e)}")
        return None


async def main():
    # Adds all images in the photos directory
    await add("photos", 'descriptor.pkl')

    # Searches for the 5 most similar images to demo.jpg
    result = await search('demo.jpg', 'descriptor.pkl', 5)

    # Opens the second most similar image and shows it
    image = open_image("photos", result[0])
    if image:
        image.show()


if __name__ == "__main__":
    asyncio.run(main())
