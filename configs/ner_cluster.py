no_entity = [0]
person1 = [1, 2]
person2 = [i for i in range(3, 11)]
region = [11]
region1 = [12, 14, 15, 16, 17, 18, 19, 20]
region2 = [13, 21, 22, 23]
weather = [24]
group = [25]
corps = [i for i in range(26, 40)]
product = [40]
product1 = [41, 42, 43, 49, 50, 51, 52, 53]
product2 = [i for i in range(44, 49)]
time = [i for i in range(54, 61)]
food = [61, 62, 63]
civil = [i for i in range(64, 72)]
finance = [72]
finance1 = [i for i in range(73, 81)]
animal = [81]
animal1 = [i for i in range(82, 91)]
plant = [91]
plant1 = [i for i in range(92, 98)]
knowledge = [i for i in range(98, 105)]
event = [105]
event1 = [i for i in range(106, 117)]
material = [i for i in range(117, 124)]
color = [i for i in range(124, 127)]
game = [i for i in range(127, 130)]
disease = [130]
count = [131]
unknown = [132]


class NER(object):

  @classmethod
  def get_info(cls):
    ret = {k: v for v, x in enumerate([cls.TIER0, cls.TIER1, cls.TIER2, cls.TIER3, cls.TIER4]) for k in x.to_list()}
    return dict(sorted(ret.items()))

  class MEnum(object):
    @classmethod
    def to_list(cls):
      members = [getattr(cls, attr) for attr in dir(cls)
                 if not callable(getattr(cls, attr)) and not attr.startswith("__")]
      return sorted([x for y in members for x in y])

  class TIER0(MEnum):
    NO_ENTITY = no_entity

  class TIER1(MEnum):
    PERSON2 = person2
    PRODUCT2 = product2
    FOOD = food
    ANIMAL1 = animal1
    PLANT1 = plant1
    GAME = game

  class TIER2(MEnum):
    REGION1 = region1
    CORPS = corps
    PRODUCT1 = product1
    FINANCE1 = finance1
    ANIMAL = animal
    PLANT = plant
    EVENT1 = event1

  class TIER3(MEnum):
    REGION2 = region2
    WEATHER = weather
    GROUP = group
    PRODUCT = product
    TIME = time
    CIVIL = civil
    FINANCE = finance
    KNOWLEDGE = knowledge
    EVENT = event
    DISEASE = disease

  class TIER4(MEnum):
    PERSON1 = person1
    REGION = region
    MATERIAL = material
    COLOR = color
    COUNT = count
    UNKNOWN = unknown