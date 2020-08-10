import json
import os


class Alphabet:
    def __init__(self, name, padflag=True, unkflag=True, keep_growing=True):
        self.name = name
        self.PAD = "</pad>"
        self.UNKNOWN = "</unk>"
        self.padflag = padflag
        self.unkflag = unkflag
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing
        self.next_index = 0
        if self.padflag:
            self.add(self.PAD)
        if self.unkflag:
            self.add(self.UNKNOWN)

    def clear(self, keep_growing=True):
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing
        self.next_index = 0

    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance):
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:
                index = self.next_index
                self.add(instance)
                return index
            else:
                if self.UNKNOWN in self.instance2index:
                    return self.instance2index[self.UNKNOWN]
                else:
                    print(self.name + " get_index raise wrong, return 0. Please check it")
                    return 0

    def get_instance(self, index):
        if index == 0:
            if self.padflag:
                print(self.name + " get_instance of </pad>, wrong?")
            if not self.padflag and self.unkflag:
                print(self.name + " get_instance of </unk>, wrong?")
            return self.instances[index]
        try:
            return self.instances[index]
        except IndexError:
            print('WARNING: '+ self.name + ' Alphabet get_instance, unknown instance, return the </unk> label.')
            return '</unk>'

    def size(self):
        return len(self.instances)

    def iteritems(self):
        return self.instance2index.items()

    def enumerate_items(self, start=1):
        if start < 1 or start >= self.size():
            raise IndexError("Enumerate is allowed between [1 : size of the alphabet)")
        return zip(range(start, len(self.instances) + 1), self.instances[start - 1:])

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def get_content(self):
        return {'instance2index': self.instance2index, 'instances': self.instances}

    def from_json(self, data):
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]

    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        saving_name = name if name else self.__name
        try:
            json.dump(self.get_content(), open(os.path.join(output_directory, saving_name + ".json"), 'w',encoding="utf-8"))
        except Exception as e:
            print("Exception: Alphabet is not saved: " % repr(e))

    def load(self, input_directory, name=None):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        loading_name = name if name else self.__name
        self.from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"),encoding="utf-8")))


