import json


def is_compressable(obj):
    if (
        hasattr(obj, '__len__') and
        not hasattr(obj, 'union') and
        not hasattr(obj, 'lower')
    ):
        return True
    return False


def get_roots(fmap):
    roots = []
    for ft in fmap.fts():
        o = fmap.get_obj(ft)
        if not is_compressable(o):
            roots.append((ft, o,))
    return roots


class FeatureMap(object):
    def __init__(self):
        self._obj_on_ft = {}
        self._ft_on_obj = {}
        self._cur_ix = -1

    def clear(self):
        self._obj_on_ft = {}
        self._ft_on_obj = {}
        self._cur_ix = -1

    def fts(self):
        return set(self._obj_on_ft.keys())

    def _compress(self, obj):
        tup = []
        for item in obj:
            tup.append(self.get_ft(item))
        return tuple(tup)

    def _decompress(self, obj):
        tup = []
        for ftnum in obj:
            tup.append(self.get_obj(ftnum))
        return tuple(tup)

    def get_obj(self, ft):
        """
        :type ft: int
        :rtype: obj
        """
        obj = self._obj_on_ft.get(ft)
        if is_compressable(obj):
            return self._decompress(obj)
        return obj

    def root_objs(self):
        return set([t[1] for t in get_roots(self)])

    def get_ft(self, obj):
        """
        :type obj: obj
        :rtype: int
        """
        if is_compressable(obj):
            return self._ft_on_obj.get(self._compress(obj))
        return self._ft_on_obj.get(obj)

    def is_ft_compressed(self, ft):
        return is_compressable(self.get_obj(ft))

    def _add_compressable_obj(self, obj):
        tup = []
        for item in obj:
            tup.append(self.get_ft_add_on_absent(item))
        tup = tuple(tup)
        self._cur_ix += 1
        self._obj_on_ft[self._cur_ix] = tup
        self._ft_on_obj[tup] = self._cur_ix

    def get_ft_add_on_absent(self, obj):
        """
        :type obj: obj
        :rtype: int
        :returns: integer mapped to object
        """
        if is_compressable(obj):
            self._add_compressable_obj(obj)
            return int(self._cur_ix)

        ft = self.get_ft(obj)
        if ft is not None:
            return ft

        self._cur_ix += 1
        self._obj_on_ft[self._cur_ix] = obj
        self._ft_on_obj[obj] = self._cur_ix
        return int(self._cur_ix)

    def num_fts(self):
        return self._cur_ix + 1


def rm_objs_remap(fmap, rmfts):
    # remove compressed features first
    rmfts = set(rmfts)
    newfmap = FeatureMap()
    for ft, obj in fmap._obj_on_ft.items():
        if ft in rmfts:
            continue
        o = fmap.get_obj(ft)
        newfmap.get_ft_add_on_absent(o)
    return newfmap


def dumps(fmap):
    return json.dumps(fmap._obj_on_ft)


def loads(dta):
    fm = FeatureMap()
    obj_on_ft = json.loads(dta)
    obj_on_ft = {int(ft): obj for ft, obj in obj_on_ft.items()}
    for ft in obj_on_ft.keys():
        if is_compressable(obj_on_ft[ft]):
            obj_on_ft[ft] = tuple(obj_on_ft[ft])
    ft_on_obj = {v: k for k, v in obj_on_ft.items()}
    fm._cur_ix = max(obj_on_ft.keys())
    fm._obj_on_ft = obj_on_ft
    fm._ft_on_obj = ft_on_obj
    return fm
