namespace DeepLearning
{
	void *_std_allocate(size_t _Count, size_t _Sz)
	{        // allocate storage for _Count elements of size _Sz
		void *_Ptr = 0;
		const size_t _User_size = _Count * _Sz;
		_Ptr = ::operator new(_User_size);
		return _Ptr;
	}

	void _std_deallocate(void * _Ptr, size_t _Count, size_t _Sz)
	{        // deallocate storage for _Count elements of size _Sz
		::operator delete(_Ptr);
	}
}