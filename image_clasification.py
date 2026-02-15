import os


def main():
    folder = "image_parking"
    if not os.path.isdir(folder):
        print(f"The folder '{folder}' does not exist.")
        return

    for count, filename in enumerate(os.listdir(folder)):
        dst = f"{str(count).zfill(5)}.jpg"
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, dst)
        try:
            # Rename the file
            os.rename(src, dst)
            print(f"Renamed '{src}' to '{dst}'")
        except Exception as e:
            print(f"Error renaming '{src}' to '{dst}': {e}")


if __name__ == '__main__':
    main()