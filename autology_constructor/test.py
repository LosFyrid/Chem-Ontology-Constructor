from owlready2 import *
import owlready2.base
import owlready2.class_construct

from config.settings import ONTOLOGY_CONFIG

onto = get_ontology("data/ontology/backup-2.owl").load()
#onto = ONTOLOGY_CONFIG["ontology"]

class_namespace = ONTOLOGY_CONFIG["classes"]
data_property_namespace = ONTOLOGY_CONFIG["data_properties"]
object_property_namespace = ONTOLOGY_CONFIG["object_properties"]

classes_names = sorted([cls.name for cls in onto.classes() if isinstance(cls, ThingClass)])

object_properties_names = sorted([prop.name for prop in onto.object_properties()])
data_properties_names = sorted([prop.name for prop in onto.data_properties()])

print(classes_names)
print(object_properties_names)
print(data_properties_names)
print("--------------------------------")

object_properties = list(onto.object_properties())
data_properties = list(onto.data_properties())

print(object_properties)
print(data_properties)
print("--------------------------------")

test_object_property = object_properties[0]
test_data_property = data_properties[0]

test_class = class_namespace["AE-C(4)Ps"]
super_classes = test_class.is_a
ancestors = test_class.ancestors()

super_class = super_classes[0]
print(super_class)
print(type(super_class))
print("--------------------------------")

restrictions = [restriction for restriction in super_classes if isinstance(restriction, owlready2.class_construct.Restriction)]
if len(restrictions) > 0:
    # for restriction in restrictions:
    #     restriction_value = restriction.__getattr__("value")
    #     restriction_is_a = restriction.is_a
    #     restriction_dict = restriction.__dict__
    #     restriction_subclasses = restriction.subclasses()
    #     restriction_property = restriction.property
    #     print(restriction_value)
    #     print(type(restriction_value))
    #     print(restriction_is_a)
    #     print(restriction_dict)
    #     print(restriction_subclasses)
    #     print(restriction_property)
    #     print(type(restriction_property))
    #     print("--------------------------------")
    restrictions = restrictions[1]
    restriction_value = restrictions.__getattr__("value")
    restriction_is_a = restrictions.is_a
    restriction_dict = restrictions.__dict__
    restriction_subclasses = restrictions.subclasses()
    restriction_property = restrictions.property
    print(restriction_value)
    print(type(restriction_value))
    print(restriction_is_a)
    print(restriction_dict)
    print(restriction_subclasses)
    print(restriction_property)
    print(type(restriction_property))
    print(type(restriction_value.Classes[0]))
    print([cls.name for cls in restriction_value.Classes])
    print([class_namespace[name] for name in [cls.name for cls in restriction_value.Classes]])
    print("--------------------------------")
else:
    restriction = None
    print("No restrictions found")






print("Super Classes Information:")
for i, sc in enumerate(super_classes):
    print(f"Super Class {i+1}:")
    print(f"  Name: {getattr(sc, 'name', 'No name attribute')}")
    print(f"  Type: {type(sc)}")
print("--------------------------------")
for i, ancestor in enumerate(ancestors):
    print(f"Ancestor {i+1}:")
    print(f"  Name: {getattr(ancestor, 'name', 'No name attribute')}")
    print(f"  Type: {type(ancestor)}")
print("--------------------------------")







'''
print("查找所有Restriction对象:")
print("--------------------------------")

# 获取本体中的所有类
all_classes = list(onto.classes())
print(f"本体中共有 {len(all_classes)} 个类")

# 存储所有找到的限制条件
all_restrictions = []

# 遍历所有类
for cls in all_classes:
    # 获取类的所有超类
    cls_superclasses = cls.is_a
    
    # 筛选出限制条件
    cls_restrictions = [r for r in cls_superclasses if isinstance(r, owlready2.class_construct.Restriction)]
    
    # 如果找到限制条件，添加到结果列表
    if cls_restrictions:
        all_restrictions.extend(cls_restrictions)
        print(f"类 '{getattr(cls, 'name', str(cls))}' 有 {len(cls_restrictions)} 个限制条件")

# 去重（因为不同类可能使用相同的限制条件）
unique_restrictions = list(set(all_restrictions))

print(f"\n总共找到 {len(all_restrictions)} 个限制条件，去重后有 {len(unique_restrictions)} 个")
print("--------------------------------")

# 输出每个限制条件的详细信息
for i, restriction in enumerate(unique_restrictions[:10]):  # 只显示前10个，避免输出过多
    print(f"限制条件 {i+1}:")
    print(f"  类型: {type(restriction)}")
    print(f"  属性: {restriction.property}")
    print(f"  值类型: {getattr(restriction, 'value', None)}")
    print(f"  字典内容: {restriction.__dict__}")
    print()

if len(unique_restrictions) > 10:
    print(f"... 还有 {len(unique_restrictions) - 10} 个限制条件未显示")

print("--------------------------------")
print("统计限制条件的type字段值:")
print("--------------------------------")

# 创建一个字典来统计type字段的值
type_counts = {}

# 遍历所有限制条件
for restriction in unique_restrictions:
    # 获取restriction字典中的type字段
    restriction_type = restriction.__dict__.get('type')
    
    # 如果type字段存在，增加计数
    if restriction_type is not None:
        if restriction_type in type_counts:
            type_counts[restriction_type] += 1
        else:
            type_counts[restriction_type] = 1

# 输出统计结果
if type_counts:
    print(f"找到以下type字段值的统计:")
    for type_value, count in type_counts.items():
        print(f"  {type_value}: {count}个")
else:
    print("没有找到任何限制条件包含type字段")

    print("--------------------------------")
    print("查找具有多个子类的限制条件:")
    print("--------------------------------")
        
# 存储具有多个子类的限制条件及其子类
multi_subclass_restrictions = {}

# 遍历所有限制条件
for restriction in unique_restrictions:
    try:
        # 获取子类
        subclasses = list(restriction.subclasses())
        
        # 检查子类数量是否大于1
        if len(subclasses) > 1:
            multi_subclass_restrictions[restriction] = subclasses
            print(f"限制条件 {restriction} 有 {len(subclasses)} 个子类")
    except Exception as e:
        # 如果调用subclasses()方法出错，跳过该限制条件
        continue

# 输出详细信息
if multi_subclass_restrictions:
    print("\n具有多个子类的限制条件详情:")
    for i, (restriction, subclasses) in enumerate(multi_subclass_restrictions.items()):
        print(f"\n限制条件 {i+1}: {restriction}")
        print(f"  属性: {restriction.property}")
        print(f"  值类型: {getattr(restriction, 'value', None)}")
        print(f"  子类数量: {len(subclasses)}")
        print(f"  子类列表:")
        for j, subclass in enumerate(subclasses[:5]):  # 只显示前5个子类
            print(f"    {j+1}. {subclass}")
        if len(subclasses) > 5:
            print(f"    ... 还有 {len(subclasses) - 5} 个子类未显示")
else:
    print("没有找到具有多个子类的限制条件")
'''

