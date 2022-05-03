import argparse
def parseArgs():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--save_path", type=str, default="./save/")
    parser.add_argument("--data_path", type=str, default="../Data/")
    parser.add_argument("--clear", type=str, default="clear")
    parser.add_argument("--hazy", type=str, default="hazy")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--g_a_path", type=str, default="generator_a.pkl")
    parser.add_argument("--g_b_path", type=str, default="generator_b.pkl")
    parser.add_argument("--d_a_path", type=str, default="discriminator_a.pkl")
    parser.add_argument("--d_b_path", type=str, default="discriminator_b.pkl")
    args = parser.parse_args()
    return args
