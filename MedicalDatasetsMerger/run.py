from DataMerger import DataMerger
import traceback

def main():
    dm = DataMerger(constants_yml_filename="./utils/merge_constants_config.yml")
    try:
        dm.start_merging()
        dm.insert_SMILES_imgs()
    except Exception as e:
        print(traceback.format_exc())


if __name__ == '__main__':
    # TODO: parse argments
    main()
