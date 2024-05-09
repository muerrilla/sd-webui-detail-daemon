import launch

if not launch.is_installed("numpy"):
    launch.run_pip("install numpy")
    print("Installing numpy...")

if not launch.is_installed("matplotlib"):
    launch.run_pip("install matplotlib")
    print("Installing matplotlib...")